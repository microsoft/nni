import * as React from 'react';
import { DetailsRow, IDetailsRowBaseProps } from '@fluentui/react';
import OpenRow from '../../public-child/OpenRow';

interface DetailsProps {
    detailsProps: IDetailsRowBaseProps;
}

interface DetailsState {
    isExpand: boolean;
}

class Details extends React.Component<DetailsProps, DetailsState> {
    constructor(props: DetailsProps) {
        super(props);
        this.state = { isExpand: false };
    }

    render(): React.ReactNode {
        const { detailsProps } = this.props;
        const { isExpand } = this.state;
        return (
            <div>
                <div
                    onClick={(): void => {
                        this.setState(() => ({ isExpand: !isExpand }));
                    }}
                >
                    <DetailsRow {...detailsProps} />
                </div>
                {isExpand && <OpenRow trialId={detailsProps.item.id} />}
            </div>
        );
    }
}

export default Details;
