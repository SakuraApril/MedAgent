@echo off
"C:\Windows\System32\OpenSSH\ssh.exe" -p 443 -o UserKnownHostsFile="C:\Users\33272\Documents\deepseek-harness\default-workspace\.ssh_known_hosts" -o StrictHostKeyChecking=accept-new %*
