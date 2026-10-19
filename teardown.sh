#!/usr/bin/env bash
# Deletes everything deploy.sh created: the CloudFront distribution, the Origin
# Access Control, and the S3 bucket with its files. The site goes offline for good.
set -euo pipefail

NAME="digit-recognizer"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET="$NAME-$ACCOUNT_ID"

read -r -p "Permanently delete the $NAME site and bucket $BUCKET? [y/N] " answer
[ "$answer" = "y" ] || exit 1

DIST_ID=$(aws cloudfront list-distributions \
  --query "DistributionList.Items[?Comment=='$NAME'].Id | [0]" --output text)
if [ "$DIST_ID" != "None" ]; then
  # A distribution must be disabled (and finish deploying that change) before it can be deleted.
  ETAG=$(aws cloudfront get-distribution-config --id "$DIST_ID" --query ETag --output text)
  aws cloudfront get-distribution-config --id "$DIST_ID" --query DistributionConfig \
    | sed 's/"Enabled": true/"Enabled": false/' > /tmp/$NAME-dist.json
  aws cloudfront update-distribution --id "$DIST_ID" --if-match "$ETAG" \
    --distribution-config file:///tmp/$NAME-dist.json >/dev/null
  echo "==> Disabling distribution (can take several minutes)..."
  aws cloudfront wait distribution-deployed --id "$DIST_ID"
  ETAG=$(aws cloudfront get-distribution --id "$DIST_ID" --query ETag --output text)
  aws cloudfront delete-distribution --id "$DIST_ID" --if-match "$ETAG"
fi

OAC_ID=$(aws cloudfront list-origin-access-controls \
  --query "OriginAccessControlList.Items[?Name=='$NAME'].Id | [0]" --output text)
if [ "$OAC_ID" != "None" ]; then
  ETAG=$(aws cloudfront get-origin-access-control --id "$OAC_ID" --query ETag --output text)
  aws cloudfront delete-origin-access-control --id "$OAC_ID" --if-match "$ETAG"
fi

aws s3 rb "s3://$BUCKET" --force 2>/dev/null || true   # --force empties the bucket first
echo "Done."
