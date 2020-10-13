import * as React from 'react';

interface LogpathChildProps {
    eachLogpath: string;
    logName: string;
}

class LogPathChild extends React.Component<LogpathChildProps, {}> {
    constructor(props: LogpathChildProps) {
        super(props);
    }

    render(): React.ReactNode {
        const { eachLogpath, logName } = this.props;
        const isLink = /^http/gi.test(eachLogpath);

        return (
            <div className='logpath'>
                <span className='logName'>{logName}</span>
                {isLink ? (
                    <a className='logContent logHref' rel='noopener noreferrer' href={eachLogpath} target='_blank'>
                        {eachLogpath}
                    </a>
                ) : (
                    <span className='logContent'>{eachLogpath}</span>
                )}
            </div>
        );
    }
}

export default LogPathChild;
