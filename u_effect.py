from fireworks import FiretaskBase, FWAction, explicit_serialize




@explicit_serialize
class analyzeUEffect(FiretaskBase):
    """We retrieve the relaxed slab at the +U and use that as a
    starting point for the adslab relaxations, with rotational sweep at
    that same value of U"""

    required_params = [
    ]

    def run_task(self, fw_spec):
        print(f"{fw_spec}")
        return FWAction()
