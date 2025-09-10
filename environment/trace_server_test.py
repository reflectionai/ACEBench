#!/usr/bin/env python3
"""Test script for Reset and Step RPCs."""

import asyncio
import grpc
import pytest
from proto.generated import env_rpc_pb2, env_rpc_pb2_grpc, trace_pb2


import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@pytest.mark.asyncio
async def test_reset_and_step(test_num: int):
    """Test both Reset and Step RPCs in sequence."""
    async with grpc.aio.insecure_channel("127.0.0.1:50051") as channel:
        stub = env_rpc_pb2_grpc.EnvRpcStub(channel)

        # Test Reset first
        print("=== Testing Reset ===")
        ace_info = env_rpc_pb2.AceBenchTask(
            model_name="gpt-4o",
            test_category="normal_single_turn_single_function",
            test_number=test_num,
            temperature=0.0,
        )

        reset_request = env_rpc_pb2.ResetRequest()
        trace_info = reset_request.trace.Extensions[env_rpc_pb2.ace_bench_trace_info]
        trace_info.task.CopyFrom(ace_info)

        reset_response = await stub.Reset(reset_request)
        assert reset_response is not None
        assert reset_response.trace is not None

        # Test Step after successful reset
        print("=== Testing Step ===")
        message = trace_pb2.Message(
            role=trace_pb2.Role.ASSISTANT,
            text="[InheritanceLegalAdvisor_queryInheritanceLaws(deceased={'nationality': 'Indian', 'countryOfResidence': 'US', 'dateOfDeath': '2023-11-15'}, beneficiaries=[{'name': 'user', 'relationship': 'grandchild', 'countryOfResidence': 'Canada'}, {'name': 'sister', 'relationship': 'grandchild', 'countryOfResidence': 'India'}], legalIssues=['taxation', 'property transfer', 'will validation'], consultationDate='2023-12-05')]",
        )
        trace: trace_pb2.Trace = reset_response.trace
        trace.messages.append(message)
        step_request = env_rpc_pb2.StepRequest(trace=trace)
        step_response = await stub.Step(step_request)
        assert step_response is not None
        logger.info("\t step_response:", step_response)


# Keep the direct run capability too
async def main():
    for i in range(3, 5):
        await test_reset_and_step(i)


if __name__ == "__main__":
    asyncio.run(main())
