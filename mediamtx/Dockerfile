# MediaMTX + ffmpeg (for the on-demand demo publisher in mediamtx.yml).
ARG MEDIAMTX_VERSION=latest
FROM bluenviron/mediamtx:${MEDIAMTX_VERSION} AS mediamtx

FROM alpine:3.20
RUN apk add --no-cache ffmpeg
COPY --from=mediamtx /mediamtx /mediamtx
ENTRYPOINT ["/mediamtx"]
CMD ["/mediamtx.yml"]
