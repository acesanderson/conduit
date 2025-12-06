from conduit.storage.odometer.odometer_base import Odometer


class SessionOdometer(Odometer):
    """
    Attaches to Model as a singleton (._session_odomoter).
    Always loads by default.
    """
