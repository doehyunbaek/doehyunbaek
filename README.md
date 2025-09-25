# Personal utilities

## Commands

### devc_to_docker

Capture a running VS Code dev container (including bind-mounted workspace files) into a reusable Docker image.

```bash
doehyunbaek devc_to_docker <container-id>
```

Key options:

- `-o, --output-image`: tag for the final snapshot image (defaults to `devc_to_docker/<container>:<timestamp>`)
- `--intermediate-image`: custom tag for the intermediate image created during the process
- `--temp-root`: directory under which temporary workspace archives are placed (default: system `/tmp`)
- `--keep-temp`: retain the temporary workspace copy instead of deleting it
- `--keep-intermediate`: keep the intermediate image instead of removing it

Requirements: Docker CLI installed locally and access to the target container.
