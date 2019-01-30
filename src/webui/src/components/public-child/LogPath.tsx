import * as React from 'react';
import LogPathChild from './LogPathChild';

interface LogpathProps {
    logStr: string;
}

class LogPath extends React.Component<LogpathProps, {}> {

    constructor(props: LogpathProps) {
        super(props);

    }

    render() {
        const { logStr } = this.props;
        const isTwopath = logStr.indexOf(',') !== -1
            ?
            true
            :
            false;
        return (
            <div>
                {
                    isTwopath
                        ?
                        <div>
                            <LogPathChild
                                eachLogpath={logStr.split(',')[0]}
                                logName="LogPath:"
                            />
                            <LogPathChild
                                eachLogpath={logStr.split(',')[1]}
                                logName="hdfsLogPath:"
                            />
                        </div>
                        :
                        <LogPathChild
                            eachLogpath={logStr}
                            logName="LogPath:"
                        />
                }
            </div>
        );
    }
}

export default LogPath;