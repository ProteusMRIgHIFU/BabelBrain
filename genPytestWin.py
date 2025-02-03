from subprocess import check_output, CalledProcessError, STDOUT


if __name__ == "__main__":
    command =['pytest', "-k","test_steps_normal and brainsight and Deep_Target and ( -CT- or -NONE- or -ZTE- ) and ID_0082","--co"]
    try:
        output = check_output(command, stderr=STDOUT).decode()
        success = True 
    except CalledProcessError as e:
        output = e.output.decode()
        success = False

    print('return code', success)
    if (not success):
        print(output)
        raise SystemError("Unable to generate list of cases")

    ListTx=[]
    for l in output.splitlines():
        if 'Deep_Target-ID_0082-' in l:
            tx=l.split('Deep_Target-ID_0082-')[1].split(']')[0]
            if tx not in ListTx:
                ListTx.append(tx)
    print(ListTx)

    with open('runPytestWin.bat','w') as f:
        cmd = 'del PyTest_Reports\\*.html\n'
        f.write(cmd)
        for imgtype in ['-CT-','-NONE-', '-ZTE-']:
            for tx in ListTx:
                cmd='pytest -k "test_steps_normal and brainsight and Deep_Target and ID_0082 and ' + imgtype + ' and ' + tx +'"\n'
                f.write(cmd)
                cmd = 'if %errorlevel% neq 0 exit /b %errorlevel%\n'
                f.write(cmd)
    