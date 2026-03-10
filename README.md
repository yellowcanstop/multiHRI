### Prerequisites
Download Docker Desktop.
- Windows users: Ensure the "WSL 2 Backend" is enabled.
- Mac users: Ensure "VirtioFS" is enabled in Settings > General

VS Code Extensions: Install the "Dev Containers" extension.

### Setup
```
git clone --recursive https://github.com/yellowcanstop/multiHRI.git
cd multiHRI
```
Note: If you already cloned the repo without --recursive, run git submodule update --init --recursive to fetch the missing tarware and overcooked submodules.

Open in VS Code. 
Notification: "Folder contains a Dev Container configuration. Reopen in Container."
Click Reopen in Container.

## Train
`python -m scripts.train_agents --algo-name SP --pop-total-training-timesteps 100 --n-envs 1 --epoch-timesteps 100`

## TODO
1. train agent for tarware. check eval code.
2. tarware env.render() -> save as videofile
