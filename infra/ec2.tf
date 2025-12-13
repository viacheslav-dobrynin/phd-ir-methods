resource "aws_instance" "experiments" {
  ami           = var.ami_id
  instance_type = var.instance_type
  key_name      = aws_key_pair.deployer.key_name

  tags = {
    Name = "ExperimentsInstance"
  }
}

resource "aws_ebs_volume" "experiments_data" {
  availability_zone = aws_instance.experiments.availability_zone
  size              = var.data_volume_size
  type              = var.data_volume_type

  tags = {
    Name = "ExperimentsData"
  }
}

resource "aws_volume_attachment" "experiments_data" {
  device_name = var.data_volume_device_name
  volume_id   = aws_ebs_volume.experiments_data.id
  instance_id = aws_instance.experiments.id
}
