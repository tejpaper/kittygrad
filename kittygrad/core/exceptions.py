# TODO: exceptions hierarchy, partial utils migration

class InplaceModificationError(RuntimeError):
    def __init__(self, message: str = "One of the variables needed for gradient computation "
                                      "has been modified by an inplace operation.",
                 ) -> None:
        super().__init__(message)


class RedundantBackwardError(RuntimeError):
    def __init__(self, message: str = "Trying to backward through the graph a second time.") -> None:
        super().__init__(message)


class TruncatedGraphError(RuntimeError):
    def __init__(self, message: str = "Visualization of the computational graph "
                                      "must be built starting from the leaves.",
                 ) -> None:
        super().__init__(message)
