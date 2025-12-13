variable "region" {
  type        = string
  description = "AWS region"
  default     = "us-east-1"
}

variable "ssh_public_key" {
  type        = string
  description = "SSH public key"
}

variable "ami_id" {
  type        = string
  description = "AMI ID"
  default     = "ami-0b58220a5f99bd460" # Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) (64-bit (x86))
}

variable "instance_type" {
  type        = string
  description = "EC2 instance type"
  default     = "t3.micro"
}

variable "data_volume_size" {
  type        = number
  description = "Size (in GiB) of the additional data EBS volume"
  default     = 100
}

variable "data_volume_type" {
  type        = string
  description = "EBS volume type for the data volume"
  default     = "gp3"
}

variable "data_volume_device_name" {
  type        = string
  description = "Device name used to attach the data volume"
  default     = "/dev/sdf"
}
