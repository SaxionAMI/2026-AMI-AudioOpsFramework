import os
import glob
from pathlib import Path
from sys import prefix
from dagster import asset, AssetKey, AssetIn, Output, MetadataValue

# We import these here to keep them out of the main definitions startup
from audio_ops_dagster.core.audio_processor import AudioProcessor
from audio_ops_dagster.core.audio_dataset import AudioDataset
from audio_ops_dagster.core.model_trainer import ModelTrainer
from audio_ops_dagster.registry import get_task
from torch.utils.data import DataLoader

def build_audio_assets(task_name: str, resource_key: str):
    """Factory to create assets namespaced by task_name."""
    prefix = task_name

    @asset(
        name=f"{prefix}_raw_audio_files",
        required_resource_keys={resource_key},
        group_name=prefix
    )
    def raw_audio_files(context):
        # Access the specific resource instance for this task
        config = getattr(context.resources, resource_key)
        Path(config.input_dir).mkdir(parents=True, exist_ok=True)
        return glob.glob(f"{config.input_dir}/**/*.wav", recursive=True)

    @asset(
        name=f"{prefix}_segmented_features",
        ins={"raw_audio_files": AssetIn(key=AssetKey(f"{prefix}_raw_audio_files"))},
        required_resource_keys={resource_key},
        group_name=prefix
    )
    def segmented_features(context, raw_audio_files):
        config = getattr(context.resources, resource_key)
        processor = AudioProcessor(feature_type=config.feature_type)
        feature_out = os.path.abspath(os.path.join(config.output_dir, "features"))
        return processor.process_directory(config.input_dir, feature_out)

    @asset(
        name=f"{prefix}_dataset_splits",
        ins={"segmented_features": AssetIn(key=AssetKey(f"{prefix}_segmented_features"))},
        required_resource_keys={resource_key},
        group_name=prefix
    )
    def dataset_splits(context, segmented_features):
        config = getattr(context.resources, resource_key)
        feature_dir = os.path.join(config.output_dir, "features")
        dataset_dir = os.path.join(config.output_dir, "dataset")
        
        strategy = getattr(config, "split_strategy", "random")
        context.log.info(f"Using split strategy: {strategy}")
        
        return AudioDataset.create_splits_csv(
            feature_dir=feature_dir, 
            output_dir=dataset_dir,
            strategy=strategy
        )

    @asset(
        name=f"{prefix}_trained_model",
        ins={"dataset_splits": AssetIn(key=AssetKey(f"{prefix}_dataset_splits"))},
        required_resource_keys={resource_key},
        group_name=prefix
    )
    def trained_model(context, dataset_splits):
        import mlflow
        from datetime import datetime
        config = getattr(context.resources, resource_key)
        task_config = config.to_task_config()
        
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
        experiment_name = f"Audio_Ops_{prefix.capitalize()}"
        mlflow.set_experiment(experiment_name)
               
        feature_root = os.path.join(config.output_dir, "features")
        
        train_ds = AudioDataset(csv_path=dataset_splits['train'], feature_dir=feature_root)
        val_ds = AudioDataset(csv_path=dataset_splits['val'], feature_dir=feature_root)
        
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size)
        
        task = get_task(config.task_name, task_config)
        model = task.build_model(input_shape=(1, 128, 87)) 
        
        arch = getattr(task_config, 'model_architecture', 'custom')
        feat = getattr(task_config, 'feature_type', 'spec')
        lr = getattr(task_config, 'learning_rate', 0.001)
        bs = config.batch_size # Assuming config is the Dagster resource
        
        timestamp = datetime.now().strftime("%H%M%S")
        run_name = f"{prefix.upper()}_{arch}_{feat}_lr{lr}_bs{bs}_{timestamp}"
        
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "dagster_run_id": context.run_id,
                "model_arch": arch,
                "feature_type": feat
            })
            
            context.log.info(f"Starting MLflow Run: {run_name} in Experiment: {experiment_name}")
            trainer = ModelTrainer(task, task_config, checkpoint_dir=os.path.join(config.output_dir, "checkpoints"))
            checkpoint_path = trainer.train(model=model, train_loader=train_loader, val_loader=val_loader)
            
        return {"checkpoint_path": checkpoint_path}
    
    @asset(
        name=f"{prefix}_model_evaluation",
        ins={
            "trained_model_info": AssetIn(key=AssetKey(f"{prefix}_trained_model")),
            "dataset_splits": AssetIn(key=AssetKey(f"{prefix}_dataset_splits"))
        },
        required_resource_keys={resource_key},
        group_name=prefix
    )
    def model_evaluation(context, trained_model_info, dataset_splits):
        import mlflow
        import torch
        from datetime import datetime
        
        config = getattr(context.resources, resource_key)
        task_config = config.to_task_config()
        checkpoint_path = trained_model_info["checkpoint_path"]
        
        # 1. Setup MLflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment(f"Audio_Ops_{prefix.capitalize()}")

        # 2. Reconstruct Model and Data
        task = get_task(config.task_name, task_config)
        
        # Note: input_shape could eventually move to task_config for true generic scaling
        model = task.build_model(input_shape=(1, 128, 87)) 
        model.load_state_dict(torch.load(checkpoint_path))
        
        feature_root = os.path.join(config.output_dir, "features")
        test_ds = AudioDataset(csv_path=dataset_splits['test'], feature_dir=feature_root)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size)


        # 4. Perform Evaluation using the new trainer.test() method
        # We use a run name that links it back to the specific attempt
        eval_run_name = f"{prefix.upper()}_EVAL_{datetime.now().strftime('%H%M%S')}"
        
        with mlflow.start_run(run_name=eval_run_name) as active_run:
            current_run_id = active_run.info.run_id
            
            mlflow.set_tags({
                "dagster_run_id": context.run_id,
                "mode": "evaluation",
                "split_strategy": getattr(task_config, "split_strategy", "random")
            })
            
            trainer = ModelTrainer(task, task_config)
            test_loss, metrics = trainer.test(model=model, test_loader=test_loader)
            
        # 5. Return Output with Metadata
        # Now current_run_id is safely stored even though the MLflow run is closed
        return Output(
            value=metrics,
            metadata={
                "test_loss": MetadataValue.float(test_loss),
                "mlflow_run_id": MetadataValue.text(current_run_id),
                "split_strategy": MetadataValue.text(getattr(task_config, "split_strategy", "random")),
                "checkpoint_used": MetadataValue.text(str(checkpoint_path))
            }
        )

    return [raw_audio_files, segmented_features, dataset_splits, trained_model, model_evaluation]