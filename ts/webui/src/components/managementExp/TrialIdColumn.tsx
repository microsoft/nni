import * as React from 'react';

interface TrialIdColumnProps {
    port: number;
    id: string;
    status: string;
}

class TrialIdColumn extends React.Component<TrialIdColumnProps, {}> {
    constructor(props: TrialIdColumnProps) {
        super(props);
    }

    render(): React.ReactNode {
        const { port, id, status } = this.props;
        const hostname = window.location.hostname;
        const protocol = window.location.protocol;
        const webuiPortal = `${protocol}//${hostname}:${port}/oview`;
        return (
            <div className='succeed-padding ellipsis'>
                {status === 'STOPPED' ? (
                    <div>{id}</div>
                ) : (
                    <a href={webuiPortal} className='link' target='_blank' rel='noopener noreferrer'>
                        {id}
                    </a>
                )}
            </div>
        );
    }
}

export default TrialIdColumn;
