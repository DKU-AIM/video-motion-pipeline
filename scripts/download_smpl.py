#!/usr/bin/env python3
"""Download authorized SMPL v1.1.0 directly on the Linux server; no saved credentials."""
import argparse
import getpass
import hashlib
from http.cookiejar import CookieJar
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import warnings
from urllib.parse import urlencode
from urllib.request import build_opener, HTTPCookieProcessor, Request
import zipfile

URL = 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip'
MODEL_NAME = 'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'


def install_neutral(archive, target):
    if target.exists():
        raise FileExistsError('SMPL_NEUTRAL.pkl already exists; it will not be overwritten.')
    with zipfile.ZipFile(archive) as bundle:
        entries=[i for i in bundle.infolist() if Path(i.filename).name == MODEL_NAME and not i.is_dir() and not i.filename.startswith('__MACOSX/')]
        if len(entries) != 1:
            raise ValueError('Expected exactly one official neutral v1.1.0 model in the ZIP.')
        if entries[0].file_size > 1_000_000_000:
            raise ValueError('Unexpectedly large model entry.')
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.smpl-', delete=False) as stream:
            temporary=Path(stream.name)
            try:
                with bundle.open(entries[0]) as source:
                    shutil.copyfileobj(source, stream, length=1024*1024)
                stream.flush()
                os.fsync(stream.fileno())
                # Atomic publish with no overwrite, including against concurrent callers.
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)


def download(archive):
    if not sys.stdin.isatty():
        raise RuntimeError('Run in the Elice terminal: credentials require an interactive terminal.')
    email=input('SMPL account email: ').strip()
    with warnings.catch_warnings():
        warnings.simplefilter('error', getpass.GetPassWarning)
        password=getpass.getpass('SMPL password (hidden; not saved): ')
    if not email or not password:
        raise ValueError('Email and password are required.')
    request=Request(URL, data=urlencode({'username':email,'password':password,'commit':'Log in'}).encode())
    opener=build_opener(HTTPCookieProcessor(CookieJar()))
    archive.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(dir=archive.parent, prefix='.smpl-download-', delete=False) as stream:
        temporary=Path(stream.name)
        try:
            with opener.open(request, timeout=60) as response:
                content_type=response.headers.get('Content-Type','').lower()
                if 'html' in content_type or 'text/' in content_type:
                    raise RuntimeError('Download returned a login/error page. Check account activation and credentials; no model was installed.')
                total=0; last=time.monotonic()
                while True:
                    block=response.read(1024*1024)
                    if not block: break
                    stream.write(block); total+=len(block)
                    if time.monotonic()-last >= 5:
                        print(f'Downloaded {total/1024/1024:.1f} MiB on the server', flush=True)
                        last=time.monotonic()
            stream.flush(); os.fsync(stream.fileno())
            if not zipfile.is_zipfile(temporary):
                raise RuntimeError('Server response is not a valid ZIP; no model was installed.')
            os.link(temporary, archive)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root', type=lambda p:Path(p).expanduser().resolve(),
                        default=Path(os.environ.get('MOTION_ASSETS_ROOT','~/motion-workspace')).expanduser().resolve())
    args=parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run this on the Elice Linux server, not on your Mac.')
    root=args.assets_root
    if not (root/'ml-comotion/demo.py').is_file():
        parser.error('Prepare CoMotion first with scripts/setup_motion_demo.sh.')
    target=root/'ml-comotion/src/comotion_demo/data/smpl/SMPL_NEUTRAL.pkl'
    if target.exists():
        print('Already present:', target)
        return
    archive=root/'assets/private/SMPL_python_v.1.1.0.zip'
    if not archive.exists():
        download(archive)
    install_neutral(archive,target)
    digest=hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    metadata=root/'assets/private/smpl-source.json'
    metadata.write_text(json.dumps({'source':URL,'archive':str(archive),'model':str(target),'sha256':digest.hexdigest()},indent=2)+'\n')
    metadata.chmod(0o600)
    print('SMPL installed on the server:',target)
    print('Credentials were not saved. The model and ZIP are outside the Git repository.')


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        # Do not echo HTTP response bodies, request data, cookies or credentials.
        if isinstance(exc, (ValueError, RuntimeError, FileExistsError)):
            print(str(exc), file=sys.stderr)
        else:
            print('Download/install failed ('+type(exc).__name__+'). Check network or rerun in the server terminal.',file=sys.stderr)
        sys.exit(1)
