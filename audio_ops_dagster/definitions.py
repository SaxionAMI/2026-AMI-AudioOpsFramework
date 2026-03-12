import os
import yaml
import sys
from pathlib import Path
from typing import Optional, List
from dagster import Definitions, ConfigurableResource
import inspect
# Force path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_ops_dagster.registry import TaskConfig, discover_tasks
from audio_ops_dagster.assets import build_audio_assets
from audio_ops_dagster.jobs import build_audio_jobs

# 1. Automatic Code Discovery (finds the @register_task classes)
discover_tasks()

# 2. Resource Class definition
class AudioTaskResource(ConfigurableResource):
    task_name: str
    input_dir: str
    output_dir: str
    feature_type: str = "mel"
    num_classes: Optional[int] = None
    class_labels: Optional[List[str]] = None
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    criterion: str = "cross_entropy"
    model_architecture: str = "cnn"
    split_strategy: str = "random"

    def to_task_config(self) -> TaskConfig:
        data = self.dict()
        
        # Get what the dataclass expects
        expected_keys = set(inspect.signature(TaskConfig).parameters.keys())
        provided_keys = set(data.keys())

        # OPTIONAL: Catch typos during development
        # This will warn you if you put something in YAML that TaskConfig ignores
        ignored = provided_keys - expected_keys
        if ignored:
            # We log it so you know it's not being passed to the model
            print(f"DEBUG: The following keys are orchestration-only: {ignored}")

        filtered_data = {k: v for k, v in data.items() if k in expected_keys}
        return TaskConfig(**filtered_data)

# 3. Iterate through all task folders to build the Definitions
all_assets = []
all_jobs = []
all_resources = {}

# Locate the tasks directory
tasks_root = Path(__file__).parent / "tasks"

if tasks_root.exists():
    for task_dir in tasks_root.iterdir():
        # Look for folders containing a config.yaml
        if task_dir.is_dir() and (task_dir / "config.yaml").exists():
            task_name = task_dir.name
            resource_key = f"{task_name}_config"
            
            # Load the YAML config
            with open(task_dir / "config.yaml") as f:
                data = yaml.safe_load(f)
                # Apply defaults
                data.setdefault("input_dir", "audio_input/")
                data.setdefault("output_dir", f"storage/{task_name}/")
                
                # Create the resource instance
                res_instance = AudioTaskResource(task_name=task_name, **data)
                all_resources[resource_key] = res_instance
            
            # Build assets and jobs for this specific task
            all_assets.extend(build_audio_assets(task_name, resource_key))
            all_jobs.extend(build_audio_jobs(task_name))

# 4. Final Aggregated Definitions
defs = Definitions(
    assets=all_assets,
    jobs=all_jobs,
    resources=all_resources
)