"""Dataset loaders for the BFI privacy-utility project.

Public API:

    BeamSenseLoader     iterator over BeamSense (Haque et al., 2025)
                        post-MATLAB-extraction `.mat` batches.
    InternalCSILoader   iterator over our HSG ICV CSI mesh dataset
                        (`dataset-release/`).  TODO.
"""

from .beamsense import (
    BeamSenseLoader,
    BeamSenseRecord,
    BeamSenseSplit,
    parse_beamsense_filename,
)
from .internal_csi import (
    InternalCSILoader,
    InternalCSIRecord,
    csi_frame_to_complex,
    average_link_csi,
)
from .ke_bfi import KeBFILoader, KeBFIRecord
from .ntu_fi import NTUFiLoader, NTUFiRecord

__all__ = [
    # BeamSense
    "BeamSenseLoader",
    "BeamSenseRecord",
    "BeamSenseSplit",
    "parse_beamsense_filename",
    # Internal CSI
    "InternalCSILoader",
    "InternalCSIRecord",
    "csi_frame_to_complex",
    "average_link_csi",
    # Ke BFI presence
    "KeBFILoader",
    "KeBFIRecord",
    # NTU-Fi
    "NTUFiLoader",
    "NTUFiRecord",
]
