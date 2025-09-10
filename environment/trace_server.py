"""Example gRPC server for toolenv for testing."""

import asyncio
import logging

from enum import Enum

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
from proto.generated.trace_pb2 import Trace, Message, FinishReasonInfo
import eval_main as eval_lib

import grpc
import grpc.aio
from environment import constants
from environment import files as files_lib
from environment.inference import inference as inference_lib

logger = logging.getLogger(__name__)


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
    if not request_info.HasField("test_category"):
        raise ValueError("Missing test category in ResetRequest.")

    if request_info.test_category not in constants.SUPPORTED_CATEGORIES:
        raise ValueError("Unsupported category in ResetRequest.")

    return request_info


def get_test_case_file_path(category: str):
    """Given the category, obtain all the test file names."""
    return constants.DATA_DIRECTORY / f"{category}.json"


class TestCasePromptParams:
    """Params that formats the test case's prompt"""

    question: str
    functions: str
    time: str
    profile: str


class TestCategory(Enum):
    AGENT_MULTI_TURN = 1
    AGENT_MULTI_STEP = 2
    SINGLE_TURN_INFERENCE = 3


class EnvRpcService(EnvRpcServicer):
    """EnvRpcService."""

    model_name: str
    temperature: float = 0.0
    top_p: int = 0
    testfile_category: str
    test_number: int
    test_case: dict[str, object]
    inference: inference_lib.APIModelInference
    prompt_params: TestCasePromptParams
    test_category: TestCategory

    async def Reset(
        self,
        request: ResetRequest,
        context: grpc.aio.ServicerContext[ResetRequest, ResetResponse],
    ) -> ResetResponse:
        """Reset()."""
        try:
            request_info = verify_and_get_acebench_reset_request(request)
            self.model_name = request_info.model_name
            self.testfile_category = request_info.test_category
            if "agent" in self.testfile_category:
                if "multi_turn" in self.testfile_category:
                    self.test_category = TestCategory.AGENT_MULTI_TURN
                elif "multi_step" in self.testfile_category:
                    self.test_category = TestCategory.AGENT_MULTI_STEP
            else:
                self.test_category = TestCategory.SINGLE_TURN_INFERENCE

            # DO NOT support multiple categories.
            # mapping from category -> test file should be encoded in
            # Olympus, not here.

            self.test_case = files_lib.load_test_case(
                get_test_case_file_path(self.testfile_category),
                request_info.test_number,
            )

            if request_info.HasField("temperature"):
                self.temperature = request_info.temperature
            if request_info.HasField("top_p"):
                self.temperature = request_info.top_p
            self.inference = inference_lib.APIModelInference(
                self.model_name,
                temperature=self.temperature,
                top_p=self.top_p,
                language="en",
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
            prompt_question = self.test_case["question"]
            prompt_functions = self.test_case["function"]
            prompt_time: str = self.test_case.get("time", "")
            prompt_profile: str = self.test_case.get("profile", "")
            test_case_id: str = self.test_case["id"]
            test_category = test_case_id.rsplit("_", 1)[0]

            if self.test_category == TestCategory.SINGLE_TURN_INFERENCE:
                messages = inference_lib.get_single_inference_message(
                    test_category,
                    prompt_time,
                    prompt_functions,
                    prompt_profile,
                    prompt_question,
                )
                logger.info("sending %s to open ai ", messages)
                res = inference_lib.get_response_from_client(
                    self.inference.client, messages, self.model_name
                )
            logger.info("result: %s", res)
            trace = Trace(messages=[Message(text=res)])
            response = StepResponse(trace=trace)

            # else:
            #     initial_config = self.test_case["initial_config"]
            # involved_classes = self.test_case["involved_classes"]

            # if self.test_category == TestCategory.AGENT_MULTI_TURN:
            #     result, process_list = self.inference.multi_turn_inference(
            #             prompt_question,
            #             initial_config,
            #             prompt_functions,
            #             involved_classes,
            #             test_case_id,
            #             prompt_time,
            #         )

            # elif self.test_category == TestCategory.AGENT_MULTI_STEP:
            #     result, process_list = self.inference.multi_step_inference(
            #         prompt_question,
            #         initial_config,
            #         prompt_functions,
            #         involved_classes,
            #         test_case_id,
            #         prompt_time,
            #     )

            # Eval
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
