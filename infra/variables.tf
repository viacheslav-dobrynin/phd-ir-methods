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
