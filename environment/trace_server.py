"""Example gRPC server for toolenv for testing."""

import asyncio
import logging

from enum import Enum
from pathlib import Path
from model_eval import evaluation_helper as eval_helper
from model_eval import utils as eval_utils

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
    AceBenchTask,
    ace_bench_trace_info,
    TestResult,
    AceBenchMessageInfo,
    ace_bench_msg_info,
)
from proto.generated.trace_pb2 import Trace, Message, Role
from environment.evaluation import eval as eval_lib

import grpc
import grpc.aio
from environment import constants
from environment import files as files_lib
from environment.inference import inference as inference_lib

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def validate_and_get_acebench_reset_request(
    request: ResetRequest,
) -> AceBenchTask:
    """Validate the incoming ResetRequest and return if valid."""
    if not request.trace:
        raise ValueError("Missing trace or trace from last message.")
    if not request.trace.HasExtension(ace_bench_trace_info):
        raise ValueError("Missing ace_bench_trace_info extension in ResetRequest.")
    task_info = AceBenchTask()
    task_info.CopyFrom(request.trace.Extensions[ace_bench_trace_info].task)

    if not task_info.HasField("model_name"):
        raise ValueError("Missing model_name in ResetRequest.")
    if not task_info.HasField("test_category"):
        raise ValueError("Missing test category in ResetRequest.")

    if task_info.test_category not in constants.SUPPORTED_CATEGORIES:
        raise ValueError("Unsupported category in ResetRequest.")

    return task_info


def validate_step_request_and_get_trace(request: StepRequest) -> Trace:
    if request.trace is None:
        raise ValueError(f"The Trace is not available in the request: {request}")
    if request.trace.messages is None or len(request.trace.messages) == 0:
        raise ValueError(f"No messages available from trace in the request: {request}")
    last_message = request.trace.messages[-1]
    if last_message.role != Role.ASSISTANT:
        raise ValueError("The last message is not an assistant message.")
    return request.trace


def get_test_case_file_path(category: str):
    """Given the category, obtain all the test file names."""
    return constants.DATA_DIRECTORY / f"data_{category}.json"


def get_matching_answer(testfile_category: str, test_case_id: str):
    prompt_path = eval_utils.build_data_path(
        constants.POSSIBLE_ANSWER_DIRECTORY, testfile_category
    )
    all_answers = eval_helper.load_file(prompt_path)

    return next((a for a in all_answers if a["id"] == test_case_id), None)


def trace_message_text_to_model_result(test_case_id: str, text: str):
    # TODO: probably code up the expected schema for each category.
    return {"id": test_case_id, "result": text}


def evaluate(
    trace: Trace,
    model_output_text: str,
    model_name: str,
    testfile_category: str,
    test_case_data,
) -> Message:
    test_case_id = test_case_data["id"]
    matching_answer = get_matching_answer(testfile_category, test_case_id)
    eval_result = eval_lib.run_eval(
        model_name,
        testfile_category,
        [{"id": test_case_id, "result": model_output_text}],
        eval_lib.get_paths("en"),
        "en",
        prompt=[test_case_data],
        possible_answers=[matching_answer],
    )
    # TODO: update this manual check
    assert len(eval_result) == 1
    eval_result = eval_result[0]

    # AceBenchMessageInfo,
    # ace_bench_msg_info,
    # TODO update based on the category.
    ace_result = AceBenchMessageInfo()
    ace_result.name = test_case_id
    if "error_type" in eval_result:
        ace_result.error_type = eval_result["error_type"]
    if "accuracy" in eval_result:
        ace_result.accuracy = eval_result["accuracy"]

    message = Message(role=Role.SYSTEM, text="")
    message.Extensions[ace_bench_msg_info].CopyFrom(ace_result)
    return message


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
    test_case_data: dict[str, object]
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
            task_info = validate_and_get_acebench_reset_request(request)
            self.model_name = task_info.model_name
            self.testfile_category = task_info.test_category
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

            self.test_number = task_info.test_number
            self.test_case_data = files_lib.load_test_case(
                get_test_case_file_path(self.testfile_category),
                self.test_number,
            )

            if task_info.HasField("temperature"):
                self.temperature = task_info.temperature
            if task_info.HasField("top_p"):
                self.temperature = task_info.top_p
            self.inference = inference_lib.APIModelInference(
                self.model_name,
                temperature=self.temperature,
                top_p=self.top_p,
                language="en",
            )

            prompt_question = self.test_case_data["question"]
            prompt_functions = self.test_case_data["function"]
            prompt_time: str = self.test_case_data.get("time", "")
            prompt_profile: str = self.test_case_data.get("profile", "")
            test_case_id: str = self.test_case_data["id"]
            test_category = test_case_id.rsplit("_", 1)[0]
            if self.test_category == TestCategory.SINGLE_TURN_INFERENCE:
                messages = inference_lib.get_single_inference_message(
                    test_category,
                    prompt_time,
                    prompt_functions,
                    prompt_profile,
                    prompt_question,
                )
                proto_messages = []
                for m in messages:
                    if m["role"] == "system":
                        proto_messages.append(
                            Message(role=Role.SYSTEM, text=m["content"])
                        )
                    elif m["role"] == "user":
                        proto_messages.append(
                            Message(role=Role.USER, text=m["content"])
                        )

            response = ResetResponse()
            return response

        except grpc.RpcError:
            raise
        except (ValueError, AttributeError) as e:
            logger.error("Error in Reset: %s", e)
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, f"Invalid request: {e}"
            )
        except Exception as e:
            logger.error("Unexpected error in Reset: %s", e)
            await context.abort(grpc.StatusCode.INTERNAL, f"Internal error: {e}")

    async def Step(
        self,
        request: StepRequest,
        context: grpc.aio.ServicerContext[StepRequest, StepResponse],
    ) -> StepResponse:
        """Execute a step and return the updated trace."""
        try:
            trace = validate_step_request_and_get_trace(request)

            if self.test_category == TestCategory.SINGLE_TURN_INFERENCE:
                message = evaluate(
                    trace,
                    trace.messages[-1].text,
                    self.model_name,
                    self.testfile_category,
                    self.test_case_data,
                )
                trace.messages.append(message)

                # TODO: Support more than single turn.

            return StepResponse(trace=trace)
        except grpc.RpcError:
            raise
        except (ValueError, AttributeError) as e:
            logger.error("Error in Step: %s", e)
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, f"Invalid request: {e}"
            )
        except Exception as e:
            logger.error("Unexpected error in Step: %s", e)
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
