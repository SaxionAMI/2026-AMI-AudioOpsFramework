---
title: Used in
---

# Used in

Audio Ops was developed at the
[Saxion research group Ambient Intelligence](https://www.saxion.edu/ami), which
works on technology for safety, sport and smart industry. It has been used in
the projects below. The descriptions are of the projects themselves. Results
belong to the project partners.

<!-- MAINTAINERS: confirm the AI-ZI section (the framework's role, permission
     to name partners) before publishing. See project/publishing.md. -->

## GRASS

*Gastvrij en Respectvol gedrAg van Supporters in Stadions* (hospitable and
respectful supporter behaviour in stadiums)

GRASS studies supporter behaviour in Dutch football stadiums. The project
develops technology to measure crowd sentiment and designs interventions that
nudge supporters towards positive behaviour, using machine learning and
augmented reality.

- Partners: Saxion (research groups Hospitality and Ambient Intelligence),
  Windesheim, Heracles Almelo, FC Twente, Go Ahead Eagles, KNVB, Sport Tech
  Center Twente, Go2Sure, Sorama, Adwise and Studio MAD.
- Runs from September 2024 to September 2026, co-funded by
  [Regieorgaan SIA](https://regieorgaan-sia.nl).
- [Project page on saxion.nl](https://www.saxion.nl/onderzoek/overige-projecten/ambient-intelligence/grass---gastvrij-en-respectvol-gedrag-van-supporters-in-stadions)
- Code: [SaxionAMI/2025-RAAKMKB-GRASS](https://github.com/SaxionAMI/2025-RAAKMKB-GRASS)

In a stadium there are several audio sources at once, a lot of background
noise, and the interesting signal is how the crowd changes over a match.
Several features of Audio Ops come from this setting: RTSP ingest from multiple
microphones or acoustic cameras, event and sentiment tasks on a shared media
clock, metrics to InfluxDB and Grafana for operators, and retraining crowd
sound classifiers on newly labelled match recordings with `audio-ops run`.

The GRASS repository is a good example of a project repo. It contains only the
`sentiment` task (a ResNet18 classifier for approval, disapproval and neutral
crowd sound), the Docker stack with MinIO, InfluxDB, Grafana and MediaMTX, and a
Grafana dashboard per stream. The framework is installed as a package. See
[Project repos](../guide/project-repos.md).

## AI-ZI

*AI voor bemensbare Zorg en Industrie* (AI for human-centred care and industry)

AI-ZI works on explainable AI that helps professionals decide faster and spend
less time on administration, while they stay in charge of the decision. The
cases include summarising reports in elderly care, supporting triage staff at
GP out-of-hours services, and helping operators in small manufacturing
companies.

- Led by the Saxion research groups Ambient Intelligence (design of the AI and
  prototypes) and Technology, Health & Care (implementation), with the research
  groups on ethics and employability.
- [Article in the Saxion Technology, Health & Care magazine, May 2026 (PDF)](https://www.saxion.nl/binaries/content/assets/onderzoek/lectoraten/technology-health--care/magazine-deftechnology-health--care-mei-2026-1.pdf)

In care and industry, audio analysis has to run on local hardware and a
professional must be able to check every result. For this the framework has a
separate speech-to-text lane and a timeline that shows the transcript next to
non-verbal events, with timestamps that point back to the audio. The LLM
summary (Dutch or English) is optional. Without an LLM configured, no audio or
text leaves the machine.

!!! warning "Not a medical device"
    Audio Ops is research software. It is not a medical device and has not been
    clinically validated. Do not use it for diagnosis, triage or treatment
    decisions without a qualified professional reviewing the output.

    If you deploy it in a care setting, you are responsible for complying with
    the rules that apply, such as the EU Medical Device Regulation (when used
    for a medical purpose), the GDPR (voice recordings are personal data and
    can be health data) and the EU AI Act.

## Add your project

If you use Audio Ops in a project, open a pull request that adds a section to
this page with the project name, a short description, a public link and how
you used the framework. You can also open an issue or email
[e.m.vanwolfswinkel@saxion.nl](mailto:e.m.vanwolfswinkel@saxion.nl).

## Acknowledgements

Audio Ops was developed within applied research projects at Saxion University
of Applied Sciences, some of them co-funded by Regieorgaan SIA. We thank the
project partners for the use cases and data.
