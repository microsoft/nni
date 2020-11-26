import * as React from 'react';

interface NameColumnProps {
    port: number;
    expName: string;
}

class NameColumn extends React.Component<NameColumnProps, {}> {
    constructor(props: NameColumnProps) {
        super(props);
    }

    render(): React.ReactNode {
        const { port, expName } = this.props;
        const hostname = window.location.hostname;
        const protocol = window.location.protocol;
        const webuiPortal = `${protocol}//${hostname}:${port}/oview`;
        return (
            <div className='succeed-padding ellipsis'>
                <a href={webuiPortal} className='link' target='_blank' rel='noopener noreferrer'>
                    {expName}
                </a>
            </div>
        );
    }
}

export default NameColumn;
