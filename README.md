`CUDA_VISIBLE_DEVICES=1 python -u -m  src.train_simple`
`docker compose run --rm -e CUDA_VISIBLE_DEVICES=0 -d trajimpu python -u -m src.train_model`
`conda env update -n base -f environment.yaml`