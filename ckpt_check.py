from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

if __name__ == "__main__":
    print("=========All Variables==========")
    print_tensors_in_checkpoint_file("/home/ubuntu/chg_workspace/3dcnn/model/1900_autoencoder.ckpt", tensor_name=None, all_tensors=True, all_tensor_names=True)