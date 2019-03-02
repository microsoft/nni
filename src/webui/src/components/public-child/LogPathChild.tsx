import * as React from 'react';

interface LogpathChildProps {
    eachLogpath: string;
    logName: string;
}

class LogPathChild extends React.Component<LogpathChildProps, {}> {

    constructor(props: LogpathChildProps) {
        super(props);

    }

    render() {
        const { eachLogpath, logName } = this.props;
        const isLink = /^http/gi.test(eachLogpath);

        return (
            <div>
                {
                    isLink
                        ?
                        <div className="logpath">
                            <span className="logName">{logName}</span>
                            <a className="logContent logHref" href={eachLogpath} target="_blank">{eachLogpath}</a>
                        </div>
                        :
                        <div className="logpath">
                            <span className="logName">{logName}</span>
                            <span className="logContent">{eachLogpath}</span>
                        </div>
                }
            </div>
        );
    }
}

export default LogPathChild;
