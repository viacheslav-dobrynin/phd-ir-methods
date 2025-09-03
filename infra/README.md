# Infra Install and Destroy

## Install Infra

1. Generate ssh key:

    ```shell
    ssh-keygen -t ed25519 -a 100 - f ~/.ssh/id_ed25519_awsec2
    ```

2. Go to infra folder and run (here `g4dn.2xlarge` instance type is used):

    ```shell
    terraform apply -var="ssh_public_key=$(cat ~/.ssh/id_ed25519_awsec2.pub)" -var="instance_type=g4dn.2xlarge"
    ```

3. Connect using SSH:

    ```shell
    export VPS_DNS="$(terraform output -raw aws_instance_experiments_public_dns)"
    ssh -i ~/.ssh/id_ed25519_awsec2 "ubuntu@$VPS_DNS"
    ```

## Run experiments

Use [tools/setup.sh](../tools/setup.sh) script to run experiments.

## Destroy Infra

To destroy infra run the command:

```shell
terraform destroy -var="ssh_public_key=$(cat ~/.ssh/id_ed25519_awsec2.pub)"
```
