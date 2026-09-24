"""Keep positional source/market wrappers while forwarding every modern CLI option."""
def forward(command, arguments):
    from quantlab.cli import main
    args=list(arguments);options=[]
    for name in ('source','market',*(('limit',) if command=='fetch' else ())):
        if not args or args[0].startswith('-'):break
        options.extend(['--'+name,args.pop(0)])
    main([command,*options,*args])
