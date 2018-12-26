#!/usr/bin/env bash
docker run -it -p 9000:9000 --name simple -v $(pwd)/models/:/models/ epigramai/model-server:light --port=9000 --model_name=simple --model_base_path=/models/simple_model
# To stop: docker stop simple && docker rm simple