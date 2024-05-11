# Setup Docker
Run:
```docker-compose up -d```.
```This``` directory will then be mounted to the container at ```/workspace```. 
Vscode now works well with the container.

Run some more setup with:
```./setup_env.sh```, use ```chmod +x setup_env.sh``` if you need to make it executable.

# Visual apps
Run ```xhost +``` in your original terminal to allow the container to display GUI apps.