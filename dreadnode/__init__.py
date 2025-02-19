from dreadnode.main import DEFAULT_INSTANCE
from dreadnode.version import VERSION

configure = DEFAULT_INSTANCE.configure
shutdown = DEFAULT_INSTANCE.shutdown

span = DEFAULT_INSTANCE.span
task = DEFAULT_INSTANCE.task
run = DEFAULT_INSTANCE.run

log_metric = DEFAULT_INSTANCE.log_metric
log_param = DEFAULT_INSTANCE.log_param

__version__ = VERSION
