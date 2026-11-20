# Tool gateway

The Strands harness calls the model for inference and the tool gateway for source reads. Each
source has a gateway configuration entry that sets its permitted time window and provider.
The gateway chooses the connection and credential and returns post data and coverage to the
harness. The existing tool assigns report IDs and saves the result to the normal report and trace.

Gateway mode exposes configured source reads and local reporting. It disables the generic page
and feed readers, network link checks, and model calls to job-installed tools. Local text transforms
still run as trusted Python. Give the gateway a separate service identity and store access to keep
provider credentials outside the harness account. On the Mac the harness uses a Unix socket. On
AWS it can call the same gateway as a Lambda function using IAM.

## Local fixture run

Save this configuration as `.agent/gateway.toml`. Keep job-specific configuration and fixtures under
`.agent/`, and keep prompts, jobs, traces, and reports in their ignored locations.

```toml
[sources.Example]
provider = "fixture"
path = "example-posts.json"
max_hours = 24
```

Save a synthetic source response as `.agent/example-posts.json`. An empty fixture for checking that
the provider was reached looks like this.

```json
{"schema_version": 1, "posts": [], "posts_complete": true, "comments_complete": true}
```

Start the gateway in one terminal and run the harness from another using the same socket. Supply
your prompt from the job's private configuration or interactively.

```sh
uv sync --locked --no-editable
mkdir -p -m 700 "$HOME/.fetcher"
.venv/bin/fetcher-gateway --config .agent/gateway.toml --socket "$HOME/.fetcher/gateway.sock"
```

```sh
.venv/bin/fetcher --gateway-socket "$HOME/.fetcher/gateway.sock" --report-required --report-json "$JOB_PROMPT"
```

The socket directory must belong to the gateway account. For a separate harness account, give both
accounts access to a dedicated group, give the gateway account a socket directory owned by that
group with mode `0750`, and start the gateway with `--socket-group GROUP`. For a scheduled Mac job,
run the gateway under `launchd` with absolute configuration and socket paths, and use that socket
in the scheduled harness command. Verify, from the gateway's service context, that the Keychain
item can be read without an interactive prompt. Apple documents which Keychain is available in
each launch context. [Mac Keychains](https://developer.apple.com/documentation/technotes/tn3137-on-mac-keychains)

## Approved Devvit source

An approved Devvit app with [External Endpoints](https://developers.reddit.com/docs/capabilities/server/external-endpoints)
can expose a source read with the contract below. The managed app token authorizes the gateway to
call that app. Set up the approved app endpoint and its access to the intended community before
changing the scheduled job.

```toml
credential_store = "keychain"
keychain_service = "fetcher.gateway"

[sources.Example]
provider = "devvit"
endpoint = "https://example-aaa123-external.devvit.net/external/v1/read"
credential = "example-reddit"
max_hours = 24
```

Provision a generic password item in the gateway account's Keychain. The service is
`keychain_service`, the account is the value of `credential`, and the password is the managed app
token. The optional `keychain_path` selects a particular file-based Keychain. Rotate the token by
updating that item. Keep the endpoint and credential reference in private operator configuration.
The gateway reads the secret at the time of each request and sends it only as the authorization
header on the configured HTTPS endpoint. The response contains only the documented source fields.

The gateway posts a JSON object with `schema_version: 1`, the configured `source`, and UTC
`start_at` and `end_at` timestamps. The app returns HTTP 200 and a JSON object with these fields.

| Field | Contract |
| --- | --- |
| `schema_version` | Integer `1` |
| `posts_complete`, `comments_complete` | Booleans describing coverage of the requested window |
| `posts` | List of records with `title`, `text`, an ISO 8601 `published` timestamp with offset, a unique HTTP(S) permalink in `link`, and a `comments` list |
| Optional post fields | `author` and `url` for the linked page |
| Each comment | A `text` string and optional `author`, returned oldest first |

The app is responsible for gathering permitted posts and comments and marking incomplete coverage
when it reaches a source limit. The gateway enforces the requested window and response size, and
uses the number of returned comments for each post's comment count. Provider access errors, rate
limits, and invalid responses surface as read failures. Gateway errors returned to the harness
use fixed messages so a raw provider response or credential error cannot reach the model.

## AWS

Bundle `src/fetcher` and its runtime dependencies in a Python 3.12 Lambda deployment package. Set
the handler to `fetcher.gateway_transport.lambda_handler`, provide the private source configuration
at the path in `FETCHER_GATEWAY_CONFIG`, and set the function timeout to 30 seconds. Use
`credential_store = "secretsmanager"` and set each `credential` to the name or ARN of a secret
whose `SecretString` is the managed app token.

Give the gateway function's role `secretsmanager:GetSecretValue` on the named secrets. If they use
a customer-managed KMS key, give that role the required `kms:Decrypt` permission as well. Give
the harness role `lambda:InvokeFunction` on the specific function or alias and run the harness
with `--gateway-function FUNCTION_ARN`. Configure the harness's Ollama endpoint as usual using
`OLLAMA_HOST`. [Lambda invocation](https://docs.aws.amazon.com/lambda/latest/api/API_Invoke.html) ·
[Secrets Manager access](https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html)
