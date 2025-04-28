#!/bin/bash

	wget \
    --mirror \
    --warc-cdx \
    --page-requisites \
    --html-extension \
    --convert-links \
    --directory-prefix=. \
    --span-hosts \
    --domains=crossfireforum.org,www.crossfireforum.org \
    --random-wait \
    http://crossfireforum.org
