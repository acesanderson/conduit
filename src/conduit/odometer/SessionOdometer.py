from conduit.odometer.Odometer import Odometer


class SessionOdometer(Odometer):
    """
    Attaches to Model as a singleton (._session_odomoter).
    Always loads by default.
    """
