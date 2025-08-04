import tabulate_outs as TO
import os

test_dir = "test_outs/"
outfiles = [
        f'{test_dir}/{f}' for f in os.listdir(test_dir)
        ]


def check_pass_fail(out, *args, **kwargs):
    ener = TO.get_ener(out, *args, **kwargs)
    if 'PASS' in out:
        assert isinstance(ener, float), f"{out} should PASS" 
    elif 'FAIL' in out:
        assert ener == None, f"{out} should FAIL"


for out in outfiles:
    if 'xg' in out:
        out_type = 'xg'
    else:
        out_type = 'std'
    check_pass_fail(out, out_type=out_type)

print("All tests passed")
