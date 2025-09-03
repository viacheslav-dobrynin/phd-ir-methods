resource "aws_instance" "experiments" {
  ami           = var.ami_id
  instance_type = var.instance_type
  key_name      = aws_key_pair.deployer.key_name

  tags = {
    Name = "ExperimentsInstance"
  }
}

