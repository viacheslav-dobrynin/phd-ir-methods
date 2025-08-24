variable "region" {
  type        = string
  description = "AWS region"
  default     = "us-east-1"
}

variable "ssh_public_key" {
  type        = string
  description = "SSH public key"
}

variable "instance_type" {
  type        = string
  description = "EC2 instance type"
  default     = "t3.micro"
}
