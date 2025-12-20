`CUDA_VISIBLE_DEVICES=1 python -u -m  src.train_simple`
`docker compose run --rm -d --name train-simple-block -e CUDA_VISIBLE_DEVICES=0 trajimpu python -u -m src.train_simple`
`conda env update -n base -f environment.yaml`