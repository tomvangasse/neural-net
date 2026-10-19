#!/usr/bin/env bash
# Deploys the website in web/ to AWS: a private S3 bucket served over HTTPS by CloudFront.
#
#   browser --HTTPS--> CloudFront (CDN) --signed request (OAC)--> private S3 bucket
#
# Safe to re-run: existing resources are found and reused, so running it again
# just uploads the latest files and clears CloudFront's cache.
set -euo pipefail   # stop on any error, on unset variables, and on failures inside pipes

NAME="digit-recognizer"
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET="$NAME-$ACCOUNT_ID"   # bucket names are global across all of AWS, so add the account ID to make it unique
CACHING_OPTIMIZED="658327ea-f89d-4fab-a63d-7e88639e58f6"   # AWS-managed cache policy for static files

cd "$(dirname "$0")"

echo "==> Exporting model weights to web/weights.json"
.venv/bin/python export_weights.py

# ---- 1. S3 bucket: where the files live. Kept fully private. ----
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "==> Bucket $BUCKET already exists"
else
  echo "==> Creating bucket $BUCKET"
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" >/dev/null
fi
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# ---- 2. Origin Access Control: the identity CloudFront uses to sign its requests to S3. ----
OAC_ID=$(aws cloudfront list-origin-access-controls \
  --query "OriginAccessControlList.Items[?Name=='$NAME'].Id | [0]" --output text)
if [ "$OAC_ID" = "None" ]; then
  echo "==> Creating Origin Access Control"
  OAC_ID=$(aws cloudfront create-origin-access-control --origin-access-control-config \
    "Name=$NAME,SigningProtocol=sigv4,SigningBehavior=always,OriginAccessControlOriginType=s3" \
    --query OriginAccessControl.Id --output text)
fi

# ---- 3. CloudFront distribution: the public HTTPS URL, cached at edge locations worldwide. ----
DIST_ID=$(aws cloudfront list-distributions \
  --query "DistributionList.Items[?Comment=='$NAME'].Id | [0]" --output text)
if [ "$DIST_ID" = "None" ]; then
  echo "==> Creating CloudFront distribution (takes a few minutes to go live)"
  DIST_ID=$(aws cloudfront create-distribution --query Distribution.Id --output text --distribution-config "{
    \"CallerReference\": \"$NAME-$(date +%s)\",
    \"Comment\": \"$NAME\",
    \"Enabled\": true,
    \"DefaultRootObject\": \"index.html\",
    \"PriceClass\": \"PriceClass_100\",
    \"HttpVersion\": \"http2and3\",
    \"Origins\": { \"Quantity\": 1, \"Items\": [{
      \"Id\": \"s3\",
      \"DomainName\": \"$BUCKET.s3.$REGION.amazonaws.com\",
      \"OriginAccessControlId\": \"$OAC_ID\",
      \"S3OriginConfig\": { \"OriginAccessIdentity\": \"\" }
    }]},
    \"DefaultCacheBehavior\": {
      \"TargetOriginId\": \"s3\",
      \"ViewerProtocolPolicy\": \"redirect-to-https\",
      \"CachePolicyId\": \"$CACHING_OPTIMIZED\",
      \"Compress\": true,
      \"AllowedMethods\": { \"Quantity\": 2, \"Items\": [\"GET\", \"HEAD\"] }
    }
  }")
fi
DOMAIN=$(aws cloudfront get-distribution --id "$DIST_ID" --query Distribution.DomainName --output text)

# ---- 4. Bucket policy: allow reads only from *this* CloudFront distribution. ----
aws s3api put-bucket-policy --bucket "$BUCKET" --policy "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [{
    \"Sid\": \"AllowCloudFrontRead\",
    \"Effect\": \"Allow\",
    \"Principal\": { \"Service\": \"cloudfront.amazonaws.com\" },
    \"Action\": \"s3:GetObject\",
    \"Resource\": \"arn:aws:s3:::$BUCKET/*\",
    \"Condition\": { \"StringEquals\": {
      \"AWS:SourceArn\": \"arn:aws:cloudfront::$ACCOUNT_ID:distribution/$DIST_ID\"
    }}
  }]
}"

# ---- 5. Upload the site, then tell CloudFront to drop its cached copies. ----
echo "==> Uploading web/ to s3://$BUCKET"
aws s3 sync web/ "s3://$BUCKET" --delete --exclude ".DS_Store" --cache-control "public, max-age=300"
aws cloudfront create-invalidation --distribution-id "$DIST_ID" --paths "/*" >/dev/null

echo
echo "Deployed: https://$DOMAIN"
echo "(A brand-new distribution can take ~5 minutes before the URL starts working.)"
