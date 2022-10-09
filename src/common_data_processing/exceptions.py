"""
Set of exception for HD map data processing
"""

class DataProcessException(Exception):
    """
    Exception in datasets processing
    """
    pass


class AgentTrajectoryMinLengthThresholdException(DataProcessException):
    """
    If Agent trajectory length is not long enough, trajectory can't be processes
    """
    pass


class NoCandidateCenterlinesWereFoundException(DataProcessException):
    """
    No candidates centerlines were found for queried trajectory
    """
    pass


class InvalidLaneLengthSequencesException(DataProcessException):
    """
    Bad lane length sequence exception
    """
    pass
