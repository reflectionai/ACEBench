"""Example gRPC server for toolenv for testing."""

import asyncio
import logging


from proto.generated.env_rpc_pb2_grpc import (
    EnvRpcServicer,
    add_EnvRpcServicer_to_server,
)
from proto.generated.env_rpc_pb2 import (
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
    CloseRequest,
    CloseResponse,
    AceBenchResetRequest,
    ace_bench_reset_request_info,
)
from proto.generated.trace_pb2 import Trace, Message

import grpc
import grpc.aio
from environment import constants
from environment import files as files_lib
from environment.inference import inference as inference_lib


def verify_and_get_acebench_reset_request(
    request: ResetRequest,
) -> AceBenchResetRequest:
    """Validate the incoming ResetRequest and return if valid."""
    if not request.HasExtension(ace_bench_reset_request_info):
        raise ValueError(
            "Missing ace_bench_reset_request_info extension in ResetRequest."
        )
    request_info = AceBenchResetRequest()
    request_info.CopyFrom(request.Extensions[ace_bench_reset_request_info])

    if not request_info.HasField("model_name"):
        raise ValueError("Missing model_name in ResetRequest.")
    if not request_info.HasField("category"):
        raise ValueError("Missing category in ResetRequest.")

    if request_info.category not in constants.SUPPORTED_CATEGORIES:
        raise ValueError("Unsupported category in ResetRequest.")

    return request_info


def get_test_case_file_paths(category: str):
    """Given the category, obtain all the test file names."""
    # Support groupings.
    return [constants.DATA_DIRECTORY / category / ".json"]


class EnvRpcService(EnvRpcServicer):
    """EnvRpcService."""

    model_name: str
    temperature: float = 0.0
    top_p: float = 0.0
    category: str
    test_cases: dict[str, object]
    inference: inference_lib.APIModelInference

    async def Reset(
        self,
        request: ResetRequest,
        context: grpc.aio.ServicerContext[ResetRequest, ResetResponse],
    ) -> ResetResponse:
        """Reset()."""
        try:
            request_info = verify_and_get_acebench_reset_request(request)
            self.model_name = request_info.model_name
            self.category = request_info.category
            if request_info.HasField("temperature"):
                self.temperature = request_info.temperature
            if request_info.HasField("top_p"):
                self.temperature = request_info.top_p

            # support multiple categories.
            self.test_cases = files_lib.load_test_cases(
                get_test_case_file_paths(self.category)
            )
            response = ResetResponse()
            return response

        except grpc.RpcError:
            raise
        except (ValueError, AttributeError) as e:
            logging.error("Error in Reset: %s", e)
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, f"Invalid request: {e}"
            )
        except Exception as e:
            logging.error("Unexpected error in Reset: %s", e)
            await context.abort(grpc.StatusCode.INTERNAL, f"Internal error: {e}")

    async def Step(
        self,
        request: StepRequest,
        context: grpc.aio.ServicerContext[StepRequest, StepResponse],
    ) -> StepResponse:
        """Execute a step and return the updated trace."""
        try:
            trace: Trace
            # Create trace or use existing one
            if request.HasField("trace"):
                trace = request.trace
            else:
                trace = Trace()

            # Add server response message as expected by tests
            message: Message = trace.messages.add()
            message.role = 2  # ASSISTANT
            message.text = "Message 3"

            response = StepResponse()
            response.trace.CopyFrom(trace)
            return response

        except grpc.RpcError:
            raise
        except (ValueError, AttributeError) as e:
            logging.error("Error in Step: %s", e)
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, f"Invalid request: {e}"
            )
        except Exception as e:
            logging.error("Unexpected error in Step: %s", e)
            await context.abort(grpc.StatusCode.INTERNAL, f"Internal error: {e}")

    async def Close(
        self,
        request: CloseRequest,
        context: grpc.aio.ServicerContext[CloseRequest, CloseResponse],
    ) -> CloseResponse:
        """Close the environment."""
        return CloseResponse()


async def serve() -> None:
    """Start and run the gRPC server."""
    server = grpc.aio.server()
    add_EnvRpcServicer_to_server(EnvRpcService(), server)

    listen_addr = "127.0.0.1:50051"
    server.add_insecure_port(listen_addr)

    print(f"Starting server on {listen_addr}")
    await server.start()

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        print("Shutting down server...")
        await server.stop(grace=5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
