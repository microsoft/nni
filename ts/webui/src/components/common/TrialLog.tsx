import * as React from 'react';
import LogPathChild from './LogPathChild';

interface TrialLogProps {
    logStr: string;
    id: string;
}

class TrialLog extends React.Component<TrialLogProps, {}> {
    constructor(props: TrialLogProps) {
        super(props);
    }

    render(): React.ReactNode {
        const { logStr } = this.props;

        return (
            <div>
                <LogPathChild eachLogpath={logStr} logName='Log path:' />
            </div>
        );
    }
}

export default TrialLog;
