# TLS certificates — gitignored, never commit real certs to source control.
#
# Place two files here before running docker-compose.prod.yml:
#   cert.pem   — full-chain certificate (server cert + intermediates)
#   key.pem    — private key (chmod 600)
#
# Free certs via Certbot (run once on the EC2 host):
#   sudo certbot certonly --standalone -d api.yourdomain.com
#   sudo cp /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem nginx/certs/cert.pem
#   sudo cp /etc/letsencrypt/live/api.yourdomain.com/privkey.pem   nginx/certs/key.pem
#   sudo chown $USER:$USER nginx/certs/*.pem && chmod 600 nginx/certs/key.pem
