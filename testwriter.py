"""
https://stackoverflow.com/questions/52558062/auto-mounting-efs-on-ec2-instance

EC2 configs ->
    Region: US-east-1
    Placement group:
            - Disable for development
            - Enable ddp (type: cluster) for production
    IAM role: Ashley-Jon-Nodirbek-Ec2-S3
    Tenancy: 
            - Shared for development
            - Dedicated for testing
    File Systems:
            - fs-25d7a491 | ddp
    Edit Installs script to include ML packages
        $ sudo apt update
        $ pip3 install torch
        $ pip3 install torchvision
    Security Groups: 
        - Enable SSH from "This IP"
        - TCP All Inbound

  
- mount point: /mnt/efs/fs1
$ cd /mnt/efs/fs1
$ sudo chmod 777 .
$ git clone https://github.com/amtoney524/cs598-ccc-final-project.git
$ sudo git checkout ddp


EFS Security Group: sg-d4cc56cf
$ sudo mkdir 
$ sudo mkdir ~/ddp
$ mkdir ~/ddp
$ sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport fs-25d7a491.efs.us-east-1.amazonaws.com:/ ~/ddp
$ sudo mount -t efs [fs-XXXXXXXX]:/ /path/to/mount/dir
$ sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport fs-25d7a491.efs.us-east-1.amazonaws.com:/ /mnt/efs/fs1

Cleanup
$ sudo rm -r cs598-ccc-final-project --recursive
"""

def main():
    with open('output/testlog.txt', 'w') as f:
        for i in range(100):
            f.write(f'log: {i}\n')



if __name__ == '__main__':
    main()