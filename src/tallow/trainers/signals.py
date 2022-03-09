class StopTraining(Exception):
    def __init__(self, ctx, *args: object) -> None:
        super().__init__(*args)
        self.ctx = ctx

    pass
