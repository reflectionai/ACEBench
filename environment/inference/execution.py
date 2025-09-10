from proto.generated.trace_pb2 import Trace, Message, Role
from proto.generated.env_rpc_pb2 import AceBenchTaskInput, ace_bench_trace_info


def prompt(trace: Trace) -> Trace:
    # trace_info = AceBenchTaskInput()
    # trace.Extensions[ace_bench_trace_info].CopyFrom(trace_info)
    trace.messages.extend(
        [
            Message(
                role=Role.SYSTEM,
                text="",
            ),
            Message(
                role=Role.USER,
                text="",
            ),
        ]
    )
