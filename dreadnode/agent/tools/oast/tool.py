import os
import subprocess
import time
import uuid

import requests
from loguru import logger

from dreadnode.agent.tools import Toolset, tool_method


class OastTool(Toolset):
    """
    OAST (Out-of-Band Application Security Testing) tool for detecting blind vulnerabilities.
    """

    tool_name: str = "oast-tool"
    description: str = "Out-of-band testing for blind vulnerability detection"

    @tool_method()
    def interactsh_generate_payload(self, subdomain: str = "") -> str:
        """
        Generate an Interactsh payload URL for out-of-band testing.
        
        Args:
            subdomain: Optional subdomain prefix for the payload
            
        Returns:
            Interactsh payload URL that can be used for OAST testing
            
        Example:
            payload = interactsh_generate_payload("test")
            # Use payload in SSRF, XXE, RCE tests: http://test.abc123.oast.pro
        """
        try:
            unique_id = str(uuid.uuid4())[:8]
            
            if subdomain:
                payload = f"http://{subdomain}.{unique_id}.oast.pro"
            else:
                payload = f"http://{unique_id}.oast.pro"
            
            if not hasattr(self, '_interactsh_payloads'):
                self._interactsh_payloads = {}
            
            self._interactsh_payloads[unique_id] = {
                "payload": payload,
                "created_at": time.time(),
                "subdomain": subdomain
            }
            
            return f"Generated Interactsh payload: {payload}\nPayload ID: {unique_id}\nUse interactsh_check_interactions('{unique_id}') to check for callbacks."
            
        except Exception as e:
            return f"Error generating Interactsh payload: {e}"

    @tool_method()
    def interactsh_check_interactions(self, payload_id: str, wait_time: int = 10) -> str:
        """
        Check for interactions on a previously generated Interactsh payload.
        
        Args:
            payload_id: The payload ID returned from interactsh_generate_payload
            wait_time: Time to wait for interactions (seconds)
            
        Returns:
            Information about any detected interactions
        """
        try:
            if not hasattr(self, '_interactsh_payloads'):
                return "No payloads have been generated yet. Use interactsh_generate_payload() first."
                
            if payload_id not in self._interactsh_payloads:
                return f"Payload ID '{payload_id}' not found. Available IDs: {list(self._interactsh_payloads.keys())}"
            
            payload_info = self._interactsh_payloads[payload_id]
            payload = payload_info["payload"]
            domain = payload.split("://")[1].split("/")[0]
            
            logger.info(f"Waiting {wait_time} seconds for interactions on {domain}...")
            time.sleep(wait_time)
            
            try:
                result = subprocess.run(
                    ["dig", "+short", domain],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                dns_result = result.stdout.strip()
                
                try:
                    response = requests.get(f"http://{domain}", timeout=5)
                    http_status = response.status_code
                except:
                    http_status = "No response"
                
                interactions_found = bool(dns_result or http_status != "No response")
                
                report = f"""Interaction Report for {payload}:
Payload ID: {payload_id}
Domain: {domain}
Wait time: {wait_time} seconds

DNS Resolution: {dns_result if dns_result else 'No DNS interaction detected'}
HTTP Status: {http_status}
Interactions Detected: {'YES' if interactions_found else 'NO'}

Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(payload_info['created_at']))}
Elapsed: {int(time.time() - payload_info['created_at'])} seconds"""

                return report
                
            except subprocess.TimeoutExpired:
                return f"DNS lookup timed out for {domain} - this might indicate filtering or blocking"
                
        except Exception as e:
            return f"Error checking interactions: {e}"

    @tool_method()
    def interactsh_list_payloads(self) -> str:
        """
        List all generated Interactsh payloads and their status.
        
        Returns:
            List of all generated payloads with timestamps
        """
        try:
            if not hasattr(self, '_interactsh_payloads') or not self._interactsh_payloads:
                return "No Interactsh payloads have been generated yet."
            
            payloads_info = []
            for payload_id, info in self._interactsh_payloads.items():
                age_seconds = int(time.time() - info['created_at'])
                payloads_info.append(
                    f"ID: {payload_id}\n"
                    f"  Payload: {info['payload']}\n"
                    f"  Subdomain: {info.get('subdomain', 'none')}\n"
                    f"  Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info['created_at']))}\n"
                    f"  Age: {age_seconds} seconds"
                )
            
            return "Generated Interactsh Payloads:\n\n" + "\n\n".join(payloads_info)
            
        except Exception as e:
            return f"Error listing payloads: {e}"

    @tool_method()
    def generate_burp_collaborator_payload(self, subdomain: str = "") -> str:
        """
        Generate a Burp Collaborator-style payload for OAST testing.
        
        Args:
            subdomain: Optional subdomain prefix
            
        Returns:
            Collaborator-style payload URL
        """
        try:
            unique_id = str(uuid.uuid4())[:8]
            
            if subdomain:
                payload = f"http://{subdomain}.{unique_id}.burpcollaborator.net"
            else:
                payload = f"http://{unique_id}.burpcollaborator.net"
            
            if not hasattr(self, '_collaborator_payloads'):
                self._collaborator_payloads = {}
            
            self._collaborator_payloads[unique_id] = {
                "payload": payload,
                "created_at": time.time(),
                "subdomain": subdomain
            }
            
            return f"Generated Burp Collaborator payload: {payload}\nPayload ID: {unique_id}\nNote: This is a simulated collaborator payload for testing purposes."
            
        except Exception as e:
            return f"Error generating Collaborator payload: {e}"