output "aws_instance_experiments_public_dns" {
  value = aws_instance.experiments.public_dns
}

output "data_volume_id" {
  value = aws_ebs_volume.experiments_data.id
}

output "data_volume_device_name" {
  value = aws_volume_attachment.experiments_data.device_name
}
