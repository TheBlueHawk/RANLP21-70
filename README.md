# RANLP2021 
Submission #70 repository to the RANLP 2021 conference

## Build and run the project
### Build
`docker build -t "SADPGAN" . | tee log`

### Train SADPGAN
`docker run --gpus all --rm -v <projecthome>/Anonymous/src:/app  SADPGAN bash -c "cd /app/run && python -m run_sa_dpgan.py 1 2"`

### Train DPGAN
`docker run --gpus all --rm -v <projecthome>/Anonymous/src:/app DPGAN bash -c "cd /app/run && python -m run_dpgan.py 1 2"`

### Produce visuals
After training both SADPGAN and DPGAN, two log files should be present in the log directory. To produce the visuals their name has to be passed as an argument (without the .txt) as well as the metric ('NLL_gen' or 'NLL_div') in the following way:
`docker run --gpus all --rm -v <projecthome>/Anonymous/src:/app bash -c "cd /app/visual && python visual_metric.py 'metric' 'image_name' 'dpgan_log_filename' 'sa_dpgan_log_filename'`

## Original repo
https://github.com/williamSYSU/TextGAN-PyTorch
