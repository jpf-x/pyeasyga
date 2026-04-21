

class HasMembers:
    def __init__(self,members=None):
        self._members=members if members is not None else []
    def __getitem__(self,key):
        return self._members[key]
    def __iter__(self):
        for member in self._members:
            yield member
    @property
    def members(self):
        return self._members

class Constraints(HasMembers):
    def __init__(self,*args):
        super().__init__(*args)

    def __call__(self,*vars):
        return all(c(*vars) for c in self.members) if self.members else 1
