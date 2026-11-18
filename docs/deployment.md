# Deploy SleepSense on Render

The initial deployment is one Streamlit dashboard with a single combined-measurement predictor, configured by `render.yaml`. The same container can serve the authenticated device API instead. Use separate services if both are required; each service runs one process and listens on the host-assigned `PORT`. The combined model has inadequate screening performance and is explicitly marked experimental in the UI.

## Dashboard deployment

1. Publish the reviewed source, including `render.yaml`, `Dockerfile`, `requirements.lock`, and the bundled dataset, to the GitHub repository.
2. Connect Render to that repository. Create a Blueprint from its deployment branch, or create a Docker Web Service using the settings below.
3. Keep the initial service on the **Free** plan. The Blueprint selects Singapore, runs `APP_MODE=dashboard`, and uses `/_stcore/health` for health checks.
4. Let the image build finish. It installs the locked dependencies, downloads the pinned 6.4 MB public Kaggle file with checksum verification, and explicitly trains the combined SVM and historical lifestyle/oxygen baselines. Artifacts are baked into the image, so a restart needs neither training nor a writable persistent disk. See `docs/combined-evaluation.json` for the current form's performance; older baseline scores do not apply to it.
5. Wait for Render to report the deployment live, then verify its actual returned URL, the dashboard screening form, the example import, and the model report. Do not infer a URL from the service name.

Automatic redeployment is disabled in the Blueprint. After future source changes pass CI, trigger a reviewed deployment manually.

| Render setting | Value |
| --- | --- |
| Service type | Web Service |
| Runtime | Docker |
| Dockerfile | `./Dockerfile` |
| Build context | Repository root |
| Plan | Free, for a research demonstration |
| Region | Singapore |
| Environment | `APP_MODE=dashboard`, `PORT=10000` |
| Start command | Use the Dockerfile default |
| Health-check path | `/_stcore/health` |

The [Render Docker guide](https://render.com/docs/docker) describes Docker service creation, and the [Blueprint reference](https://render.com/docs/blueprint-spec) documents these settings.

The [Free service documentation](https://render.com/docs/free) explains current resource limits, idle spin-down, cold starts, and ephemeral filesystems. A Free service is a demonstration host, not an always-on IoT monitoring service. Session results can disappear when the process restarts or the browser session expires. Downloads should be saved before leaving the session.

## Verify the container locally

```bash
docker build -t sleepsense:deployment .
docker run --rm -p 127.0.0.1:8501:8501 sleepsense:deployment
```

Open `http://localhost:8501`. The container runs as an unprivileged user and checks the model before starting the dashboard. Docker's health check follows the selected application mode and port.

## Optional device API service

Create a separate service from the same repository/image, with these differences:

- Set `APP_MODE=api` and retain the Render-assigned port or set `PORT=10000`.
- Set a random `APNEA_API_KEY` of at least 24 characters as a Render secret. Never put its value in Git, the Blueprint, or the image.
- Use `/health` as the service's health-check path.
- Keep the key in the device/gateway configuration and send it as `Authorization: Bearer ...` to `/v2/screen`, with `spo2`, `heart_rate`, `respiratory_rate`, and `perfusion_index` from the same completed interval.

For a local API container:

```bash
export APNEA_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker run --rm -p 127.0.0.1:8000:8000 \
  -e APP_MODE=api -e PORT=8000 -e APNEA_API_KEY \
  sleepsense:deployment
```

The dashboard does not require this API service and does not send its inputs to a second backend. The optional API remains independently authenticated.

The old `/v1/screen` lifestyle route and `/v1/oximetry/screen` oxygen-pattern route remain for compatibility. They are separate research baselines, not the predictor used by the main form. `/v2/model` returns the combined model report.

## Hosted data and access

A hosted dashboard sends entered/imported measurements to the hosting service over its HTTPS connection. This application does not write submitted health records to a database or log request bodies. Manual results live in a browser session's server-side memory, and users can request downloads. The public demonstration should use example or otherwise appropriate data; private real-world use requires suitable access controls and a privacy/retention policy.

No model files can be uploaded through the app. Only the trusted model generated during the image build is loaded. The runtime uses the exact scikit-learn version from `requirements.lock`.

## Deployment checks

`python test.py` includes configuration tests and live local HTTP smoke tests for both launch modes on dynamically assigned ports, as well as the model, UI, and API tests. Run them before publishing. Hosted deployment is complete only after Render confirms success and returns a live URL.
