# The base set of tooling requirements for `pants test`.
# Run `pants generate-lockfiles --resolve=pytest` after adding new packages.

python_sources(
  name="generate",
  sources=['*.py']
)

python_requirements(
    name="reqs",
    resolve="base",
    source="requirements-base.txt",
    module_mapping={
        "google-generativeai": ["google.generativeai", "google"],
    },
)

python_requirements(
    name="reqs_vllm",
    resolve="vllm",
    source="requirements-vllm.txt"
)


python_requirement(
  name='pytest',
  resolve='pytest',
  requirements=[
    'anyio',
    'ipdb',
    'pip',
    'pytest',
    'pytest-asyncio',
    'pytest-custom_exit_code',
    'pytest-error-for-skips',
    'pytest-forked',
    'pytest-ignore-flaky',
    'pytest-timeout',
    'pytest-xdist',
    'pytest-repeat',
    'pytest-rerunfailures',
    'httpx',
  ],
)

# The base set of tooling requirements for docker commands.
python_requirement(
  name='dockerfile-parser',
  resolve='dockerfile-parser',
  requirements=['dockerfile>=3.2.0,<4'],
)

python_requirement(
  name='mypy',
  resolve='mypy',
  requirements=['mypy-protobuf'],
)

shell_sources(
  name='scripts',
)

# This environment is necessary to properly build Python PEX Docker images.
# See: https://www.pantsbuild.org/stable/docs/using-pants/environments
docker_environment(
  name='linux-py12', platform='linux_x86_64', image='python:3.12-bookworm'
)



__defaults__(
  {
    pex_binary: dict(environment='linux'),
    docker_image: dict(build_platform=['linux/amd64']),
  }
)


# file(
#   name='olympus_internal_cache_docker',
#   source='Dockerfile.olympus-internal-cache',
# )

