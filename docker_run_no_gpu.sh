DOCKER_GID=$(id -g)
DOCKER_GNAME=$(id -gn)
DOCKER_UNAME=$(id -un)
DOCKER_UID=$(id -u)
DOCKER_TAG="imperial_ipc"

docker build -f Dockerfile --tag=$DOCKER_TAG \
             --build-arg GID=$DOCKER_GID \
             --build-arg GNAME=$DOCKER_GNAME \
             --build-arg UNAME=$DOCKER_UNAME \
             --build-arg UID=$DOCKER_UID \
             .

docker run --rm -it --init \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --volume="$PWD:/submission" \
  --workdir=/submission \
  imperial_ipc