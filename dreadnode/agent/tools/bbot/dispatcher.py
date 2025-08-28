import traceback
import typing as t

from loguru import logger

if t.TYPE_CHECKING:
    from bbot import Scanner


class Dispatcher:
    """
    Enables custom hooks/callbacks on certain scan events
    """

    def set_scan(self, scan: "Scanner"):
        self.scan = scan

    async def on_start(self, scan: "Scanner"):
        logger.info(f"Scan started with ID: {scan.id}")

    async def on_finish(self, scan: "Scanner"):
        logger.info(f"Scan finished with ID: {scan.id}")

    async def on_status(self, status, scan_id):
        """
        Execute an event when the scan's status is updated
        """
        logger.debug(f"Setting scan ({scan_id}) status to {status}")

    async def catch(self, callback, *args, **kwargs):
        try:
            return await callback(*args, **kwargs)
        except Exception as e:
            logger(f"Error in {callback.__qualname__}(): {e}")
            logger(traceback.format_exc())
